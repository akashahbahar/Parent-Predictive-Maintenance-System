using MailKit.Net.Smtp;
using MimeKit;
using Microsoft.Extensions.Configuration;

namespace PredictiveMaintenanceAPI.Services
{
    public class NotificationService
    {
        private readonly IConfiguration _config;

        public NotificationService(IConfiguration config)
        {
            _config = config;
        }

        public void SendEmailAlert(string subject, string body)
        {
            var emailSettings = _config.GetSection("AlertSettings:Email");

            var message = new MimeMessage();
            message.From.Add(new MailboxAddress(emailSettings["SenderName"], emailSettings["SenderEmail"]));
            message.To.Add(new MailboxAddress("", emailSettings["RecipientEmail"]));
            message.Subject = subject;
            message.Body = new TextPart("plain") { Text = body };

            using var client = new SmtpClient();
            client.Connect(emailSettings["SmtpServer"], int.Parse(emailSettings["Port"]), false);
            client.Authenticate(emailSettings["SenderEmail"], emailSettings["Password"]);
            client.Send(message);
            client.Disconnect(true);
        }

    }
}
